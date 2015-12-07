# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np
from scipy import io

from ...utils import logger
from ..meas_info import create_info
from ..base import _BaseRaw, _mult_cal_one
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


def read_raw_set(fname, ch_fname, eog=None, preload=False, verbose=None):
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
    raw : Instance of RawSet
        A Raw object containing EEGLAB .set data.
    """
    return RawSet(fname=fname, ch_fname=ch_fname, eog=eog, preload=preload,
                  verbose=verbose)


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
    raw : Instance of RawSet
        A Raw object containing EEGLAB .set data.
    """
    def __init__(self, fname, ch_fname, eog=None, preload=False, verbose=None):
        """Read EEGLAB .set file.
        """
        basedir = op.dirname(fname)
        eeg = io.loadmat(fname, struct_as_record=False, squeeze_me=True)['EEG']

        # read the data
        data_fname = op.join(basedir, eeg.data)
        logger.info('Reading %s' % data_fname)

        last_samps = [int(eeg.xmax * eeg.srate)]

        # get info
        if ch_fname is not None:
            ch_fname = op.join(basedir, ch_fname)
        info = _get_info(eeg, eog, ch_fname)

        super(RawSet, self).__init__(
            info, preload, filenames=[data_fname], last_samps=last_samps,
            orig_format='double', verbose=verbose)
        logger.info('    Range : %d ... %d =  %9.3f ... %9.3f secs'
                    % (self.first_samp, self.last_samp,
                       float(self.first_samp) / self.info['sfreq'],
                       float(self.last_samp) / self.info['sfreq']))

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data"""
        scaling = 1e-6
        nchan = self.info['nchan']
        data_offset = self.info['nchan'] * start * 4
        data_left = (stop - start) * nchan
        # Read up to 100 MB of data at a time.
        blk_size = min(data_left, (50000000 // nchan) * nchan)

        with open(self._filenames[fi], 'rb', buffering=0) as fid:
            fid.seek(data_offset)
            # extract data in chunks
            for blk_start in np.arange(0, data_left, blk_size) // nchan:
                blk_size = min(blk_size, data_left - blk_start * nchan)
                block = np.fromfile(fid,
                                    dtype=np.float32, count=blk_size) * scaling
                block = block.reshape(nchan, -1, order='F')
                blk_stop = blk_start + block.shape[1]
                data_view = data[:, blk_start:blk_stop]
                _mult_cal_one(data_view, block, idx, cals, mult)
        return data
