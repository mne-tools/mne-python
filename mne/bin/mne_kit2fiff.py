'''
Created on Dec 27, 2012

@author: teon
sqd_params class is adapted from Yoshiaki Adachi's Meginfo2.cpp 
    and sqdread's getdata.m
pattern matching for coreg is from Tal Linzen
coreg methods are adapted from Christian Brodbeck's eelbrain.plot.coreg
RawKIT class is adapted from Denis Engemann et al.'s mne_bti2fiff.py
'''

from mne.fiff.raw import Raw
from mne import verbose
from mne.fiff.constants import FIFF
from struct import unpack
import numpy as np
from numpy import sin, cos
from scipy.optimize import leastsq
import time
import logging
import re

class sqd_params(object):
    """Extracts all the information from the sqd file."""


    def __init__(self, rawfile, lowpass=None, highpass=None):
        """
        Parameters
        ----------
        rawfile: str
            raw sqd file to be read
        lowpass: None | int
            value of lowpass filter set in MEG160
        highpass: None | int
            value of highpass filter set in MEG160
        
        """
        self.dynamic_range = 2 ** 12 / 2    # signed integer. range +/- 2048
        self.voltage_range = 5.
        self.rawfile = rawfile
        fid = open(rawfile, 'r')
        fid.seek(16)    # get offset of basic info
        basic_offset = unpack('i', fid.read(4))[0]  # integer are 4 bytes
        fid.seek(basic_offset)

        # basic info
        self.version = unpack('i', fid.read(4))[0]
        self.revision = unpack('i', fid.read(4))[0]
        self.sysid = unpack('i', fid.read(4))[0]
        self.sysname = unpack('128s', fid.read(128))[0].split('\n')[0]
        self.modelname = unpack('128s', fid.read(128))[0].split('\n')[0]
        self.nchan = unpack('i', fid.read(4))[0]
        self.chan_no = range(self.nchan)
        self.nmegchan = 157
        self.nrefchan = 3
        self.ntrigchan = 8
        self.nmiscchan = self.nchan - self.nmegchan - self.nrefchan - self.ntrigchan
        self.ch_names = {}
        self.ch_names['MEG'] = ['MEG %03d' % ch for ch in range(1, self.nmegchan + 1)]
        self.ch_names['REF'] = ['REF %03d' % ch for ch in range(1, self.nrefchan + 1)]
        self.ch_names['TRIG'] = ['TRIG %03d' % ch for ch in range(1, self.ntrigchan + 1)]
        self.ch_names['MISC'] = ['MISC %03d' % ch for ch in range(1, self.nmiscchan + 1)]
        self.ch_names['STIM'] = ['STIM 014']
        self.lowpass = lowpass
        self.highpass = highpass

        # amplifier gain
        fid.seek(112)
        amp_offset = unpack('i', fid.read(4))[0]
        fid.seek(amp_offset)
        amp_data = unpack('i', fid.read(4))[0]
        input_gain_bit = 11    # stored in Bit-11 to 12
        input_gain_mask = 6144  # (0x1800)
        # input_gain: 0:x1, 1:x2, 2:x5, 3:x10
        input_gains = [1, 2, 5, 10]
        self.input_gain = input_gains[(input_gain_mask & amp_data)
                                      >> input_gain_bit]
        # 0:x1, 1:x2, 2:x5, 3:x10, 4:x20, 5:x50, 6:x100, 7:x200
        output_gains = [1, 2, 5, 10, 20, 50, 100, 200]
        output_gain_bit = 0    # stored in Bit-0 to 2
        output_gain_mask = 7   # (0x0007)
        self.output_gain = output_gains[(output_gain_mask & amp_data)
                                        >> output_gain_bit]

        # channel sensitivities
        # only channels 0-159 requires gain. the additional channels
        # (trigger channels, audio and voice channels) are passed 
        # through unaffected

        fid.seek(80)
        sens_offset = unpack('i', fid.read(4))[0]
        fid.seek(sens_offset)
        sens = np.fromfile(fid, dtype='d', count=self.nchan * 2)
        self._sensitivities = np.reshape(sens, (self.nchan, 2))
        self.sensor_gain = np.ones(self.nchan)
        self.sensor_gain[:160] = self._sensitivities[:160, 1]

        # sampling info
        fid.seek(128)
        acqcond_offset = unpack('i', fid.read(4))[0]
        fid.seek(acqcond_offset)
        acq_type = unpack('i', fid.read(4))[0]
        if acq_type == 1:
            self.sfreq = unpack('d', fid.read(8))[0]
            _ = fid.read(4) # initialized estimate of samples 
            self.nsamples = unpack('i', fid.read(4))[0]
        else:
            raise NotImplementedError

        # data offset info    
        fid.seek(144)
        self.data_offset = unpack('i', fid.read(4))[0]

    def get_data(self):
        """returns an array with data extracted from sqd file"""
        
        fid = open(self.rawfile, 'r')
        fid.seek(self.data_offset)
        data = np.fromfile(fid, dtype='h', count=self.nsamples * self.nchan)
        data = np.reshape(data, (self.nsamples, self.nchan))
        # amplifier applies only to the sensor channels 0-159
        amp_gain = self.output_gain * self.input_gain
        self.sensor_gain[:160] = self.sensor_gain[:160] / amp_gain
        conv_factor = np.array((self.voltage_range / self.dynamic_range) *
                               self.sensor_gain, ndmin=2)
        self.x = (conv_factor * data).T



logger = logging.getLogger('mne')

class RawKIT(Raw):
    """ Raw object from KIT SQD file
        Adapted from mne_bti2fiff.py
    
    """
    @verbose
    def __init__(self, data_fname, mrk_fname, elp_fname, hsp_fname,
                 sns_fname, data=None, lowpass = None, highpass = None,
                 verbose=True):
        """
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
        logger.info('Extracting SQD Parameters from %s...' % data_fname)
        self.params = sqd_params(data_fname, lowpass = lowpass, highpass = highpass)
        self._data_file = data_fname
        self.mrk_fname = mrk_fname
        self.elp_fname = elp_fname
        self.hsp_fname = hsp_fname
        self.sns_fname = sns_fname

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
        info['lowpass'], info['highpass'] = self.params.lowpass, self.params.highpass
        info['sfreq'] = float(self.params.sfreq)
        info['chs'] = self._create_chs()
        info['nchan'] = self.params.nchan
        info['ch_names'] = self.params.ch_names['MEG'] + self.params.ch_names['STIM']
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

    def _create_chs(self):
        """creates a list of dicts of meg channels for raw.info"""
        
        logger.info('Setting channel info structure...')
        p = re.compile(r'\d,[A-Za-z]*,([\.\-0-9]+),([\.\-0-9]+),([\.\-0-9]+)')
        locs = np.array(p.findall(open(self.sns_fname).read()), dtype='float')
        # the arrangement of dimensions in current fif is y,x,z in [m].
        # this is to orient in Neuromag coordinates.
        self.chan_locs = locs[:, [1, 0, 2]] / 1000
        chs = []
        for idx, ch_info in enumerate(zip(self.params.ch_names['MEG'] +
                                          self.params.ch_names['REF'], 
                                          self.chan_locs), 1):
            ch_name, ch_loc = ch_info
            chan_info = {}
            chan_info['cal'] = 1.0 # calibration factor, default is 1
            chan_info['eeg_loc'] = None
            chan_info['logno'] = idx
            chan_info['scanno'] = idx
            chan_info['range'] = 1.0
            chan_info['unit_mul'] = 0 # default is 0 mne_manual p.273
            chan_info['ch_name'] = ch_name
            if ch_name.startswith('MEG'):
                chan_info['kind'] = FIFF.FIFFV_MEG_CH
                chan_info['coil_type'] = 6001
                chan_info['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
                chan_info['unit'] = FIFF.FIFF_UNIT_T # 112 = T
                # this has the coordinates, but the three unit vectors (mne p.273)
                chan_info['loc'] = ch_loc
                # this needs to be sorted out
                chan_info['coil_trans'] = None
            elif ch_name == 'STI 014':
                chan_info['kind'] = FIFF.FIFFV_STIM_CH
            chs.append(chan_info)
            return chs
        """create a synthetic channel"""
                
    def _get_coreg(self):
        """get transformation matrix and hsp points"""
        
        coreg_data = coreg(mrk_fname=self.mrk_fname, elp_fname=self.elp_fname,
                           hsp_fname=self.hsp_fname)
        dev_head_t = coreg_data.fit()
        return dev_head_t, coreg_data.hsp_points


class coreg:
    """
    Extracts digitizer points from file.
    Creates coreg transformation matrix from device to head coord.

    Attributes
    ----------
    mrk_points : np.array
        array of 5 points by coordinate (x,y,z) from marker measurement
    elp_points : np.array
        array of 5 points by coordinate (x,y,z) from digitizer laser point
    hsp_points : np.array
        array points by coordinate (x, y, z) from digitizer

    """
    def __init__(self, mrk_fname, elp_fname, hsp_fname):
        """
        Parameters
        ----------
        mrk_fname : str
            Path to marker avg file (saved as text form MEG160).
        elp_fname : str
            Path to elp digitizer file.
    
        """
        # marker point extraction
        self.mrk_src_path = mrk_fname
        # pattern by Tal:
        p = re.compile(r'Marker \d:   MEG:x= *([\.\-0-9]+), y= *([\.\-0-9]+), z= *([\.\-0-9]+)')
        str_points = p.findall(open(mrk_fname).read())
        self.mrk_points = np.array(str_points, dtype=float)/1000
        self.mrk_points = self.mrk_points[:,[1,0,2]]
        self.mrk_points[:, 0] *= -1
        # elp point extraction
        self.elp_src_path = elp_fname
        # pattern modified from Tal's mrk pattern:
        p = re.compile('%N\t\d-[A-Z]+\s+([\.\-0-9]+)\t([\.\-0-9]+)\t([\.\-0-9]+)')
        str_points = p.findall(open(elp_fname).read())
        self.elp_points = np.array(str_points, dtype=float)
        self.elp_points = self.elp_points[:,[1,0,2]]
        self.elp_points[:, 0] *= -1
        # hsp point extraction
        self.hsp_src_path = hsp_fname
        p = re.compile(r'//No.+\n(\d*)\t(\d)\s*')
        v = re.split(p, open(hsp_fname).read())[1:]
        hsp_points = np.fromstring(v[-1], sep='\t').reshape(int(v[0]), int(v[1]))
        self.hsp_points = []
        for idx, point in enumerate(hsp_points):
            point_dict = {}
            point_dict['coord_frame'] = FIFF.FIFFV_COORD_HEAD
            point_dict['ident'] = idx + 1
            # equivalent in value but may not be the proper constant
            point_dict['kind'] = FIFF.FIFFV_POINT_CARDINAL 
            point_dict['r'] = point
            self.hsp_points.append(point_dict)

    def fit(self, include=range(5)):
        """
        Fit the marker points to the digitizer points.
    
        Parameters
        ----------
        include : index (numpy compatible)
            Which points to include in the fit. Index should select among
            points [0, 1, 2, 3, 4].
        
        """
        def err(params):
            """calculates distance from target and estimate"""
            
            T = self.trans(*params[:3]) * self.rot(*params[3:])
            pts = T*np.vstack((self.elp_points[include].T, 
                       np.ones(len(self.elp_points[include]))))
            est = np.array(pts[:3].T)               
            tgt = np.array(self.mrk_points[include])
            return (tgt - est).ravel()

        # initial guess
        params = (0, 0, 0, 0, 0, 0)
        params, _ = leastsq(err, params)
        self.est_params = params
        # head-to-device
        T = self.trans(*params[:3]) * self.rot(*params[3:])
        # returns dev2head by applying the inverse
        return np.array(T.I)

    def trans(self, x=0, y=0, z=0):
        "MNE manual p. 95, a method for translating a matrix"
        
        m = np.matrix([[1, 0, 0, x],
                       [0, 1, 0, y],
                       [0, 0, 1, z],
                       [0, 0, 0, 1]], dtype=float)
        return m
    
    def rot(self, x=0, y=0, z=0):
        "From eelbrain.plot.coreg, a method for rotating a matrix"
        c_x=cos(x); c_y=cos(y); c_z=cos(z); s_x=sin(x); s_y=sin(y); s_z=sin(z);
        r = np.matrix([[c_y*c_z, -c_x*s_z+s_x*s_y*c_z, s_x*s_z+c_x*s_y*c_z, 0],
                       [c_y*s_z, c_x*c_z+s_x*s_y*s_z, -s_x*c_z+c_x*s_y*s_z, 0],
                       [-s_y, s_x*c_y, c_x*c_y, 0],
                       [0, 0, 0, 1]], dtype=float)
        return r