# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np
from scipy import io

from ...utils import logger
from ..meas_info import create_info
from ..base import _BaseRaw
from ..constants import FIFF


def _topo_to_sph(theta, radius):
    """Convert 2D topo coordinates to spherical.
    """
    sph_phi = (0.5 - radius) * 180
    sph_theta = -theta
    return sph_phi, sph_theta


def _sph_to_cart(azimuth, elevation, r):
    """Convert spherical to cartesian coordinates.
    """
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z


def _get_info(eeg, eog, ch_fname):
    """Get measurement info.
    """
    if not ch_fname.endswith('.locs'):
        msg = """Currently, only the .locs file format is supported.
              Please contact mne-python developers for more information"""
        raise NotImplementedError(msg)

    ch_names = np.loadtxt(ch_fname, dtype='S4', usecols=[3]).tolist()
    info = create_info(sfreq=eeg.srate, ch_names=ch_names)

    # load channel locations
    dtype = {'names': ('angle', 'radius'), 'formats': ('f4', 'f4')}
    angle, radius = np.loadtxt(ch_fname, dtype=dtype, usecols=[1, 2],
                               unpack=True)
    sph_phi, sph_theta = _topo_to_sph(angle, radius)
    sph_radius = np.ones((eeg.nbchan, ))
    x, y, z = _sph_to_cart(sph_theta / 180 * np.pi,
                           sph_phi / 180 * np.pi, sph_radius)
    if eog is None:
        eog = [idx for idx, ch in enumerate(ch_names) if ch.startswith('EOG')]

    for idx, ch in enumerate(info['chs']):
        ch['loc'][:3] = np.array([x[idx], y[idx], z[idx]])
        ch['unit'] = FIFF.FIFF_UNIT_V
        ch['coord_frame'] = FIFF.FIFFV_COORD_HEAD,
        ch['coil_type'] = FIFF.FIFFV_COIL_EEG,
        ch['kind'] = FIFF.FIFFV_EEG_CH

        if ch['ch_name'] in eog:
            ch['coil_type'] = FIFF.FIFFV_COIL_NONE
            ch['kind'] = FIFF.FIFFV_EOG_CH

    return info


def read_raw_set(fname, ch_fname, eog=None, verbose=None):
    """Read an EEGLAB .set file

    Parameters
    ----------
    fname : str
        Path to the .set file.
    ch_fname : str
        Path to the file containing channel locations. Currently,
        only .loc extension is supported.
    eog : list or tuple
        Names of channels or list of indices that should be designated
        EOG channels. If None (default), the channel names beginning with
        ``EOG`` are used.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    raw : Instance of RawSet
        A Raw object containing EEGLAB .set data.
    """
    return RawSet(fname, ch_fname, eog, verbose)


class RawSet(_BaseRaw):
    """Raw object from EEGLAB .set file.

    Parameters
    ----------
    fname : str
        Path to the .set file.
    ch_fname : str
        Path to the file containing channel locations. Currently,
        only .loc extension is supported.
    eog : list or tuple
        Names of channels or list of indices that should be designated
        EOG channels. If None (default), the channel names beginning with
        ``EOG`` are used.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    raw : Instance of RawSet
        A Raw object containing EEGLAB .set data.
    """
    def __init__(self, fname, ch_fname, eog=None, verbose=None):
        """Read EEGLAB .set file.
        """
        scaling = 1e-6
        basedir = op.dirname(fname)
        eeg = io.loadmat(fname, struct_as_record=False, squeeze_me=True)['EEG']

        # read the data
        data_fname = op.join(basedir, eeg.data)
        logger.info('Reading %s' % data_fname)
        data_fid = open(data_fname)
        data = np.fromfile(data_fid, dtype=np.float64) * scaling
        data = data.reshape((-1, eeg.nbchan)).T

        # get info
        if ch_fname is not None:
            ch_fname = op.join(basedir, ch_fname)
        info = _get_info(eeg, eog, ch_fname)

        super(RawSet, self).__init__(
            info, data, filenames=[fname], orig_format='double',
            verbose=verbose)
        logger.info('    Range : %d ... %d =  %9.3f ... %9.3f secs'
                    % (self.first_samp, self.last_samp,
                       float(self.first_samp) / self.info['sfreq'],
                       float(self.last_samp) / self.info['sfreq']))
