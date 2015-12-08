# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np
import warnings

from ...utils import logger
from ..meas_info import _empty_info
from ..base import _BaseRaw, _mult_cal_one, _check_update_montage
from ..constants import FIFF
from ...channels.montage import Montage


def _get_info(eeg, montage, eog):
    """Get measurement info.
    """
    info = _empty_info(sfreq=eeg.srate)
    info['nchan'] = eeg.nbchan

    if eog is None:
        eog = [idx for idx, ch in enumerate(info['ch_names'])
               if ch.startswith('EOG')]

    # add the ch_names and info['chs'][idx]['loc']
    path = None
    if len(eeg.chanlocs) > 0:
        ch_names, pos = [], []
        kind = 'user_defined'
        selection = np.arange(len(eeg.chanlocs))
        for chanloc in eeg.chanlocs:
            ch_names.append(chanloc.labels)
            pos.append([chanloc.X, chanloc.Y, chanloc.Z])
        montage = Montage(np.array(pos), ch_names, kind, selection)
    elif isinstance(montage, str):
        path = op.dirname(montage)
    _check_update_montage(info, montage, path=path,
                          update_ch_names=True)

    # update the info dict
    cal = 1e-6
    for idx, ch_name in enumerate(info['ch_names']):
        info['chs'][idx]['cal'] = cal
        if ch_name in eog:
            ch['coil_type'] = FIFF.FIFFV_COIL_NONE
            ch['kind'] = FIFF.FIFFV_EOG_CH

    return info


def read_raw_eeglab(fname, montage=None, eog=None, preload=False,
                    verbose=None):
    """Read an EEGLAB .set file

    Parameters
    ----------
    fname : str
        Path to the .set file.
    montage : str | None | instance of montage
        Path or instance of montage containing electrode positions.
        If None, sensor locations are (0,0,0). See the documentation of
        :func:`mne.channels.read_montage` for more information.
    eog : list or tuple
        Names of channels or list of indices that should be designated
        EOG channels. If None (default), the channel names beginning with
        ``EOG`` are used.
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory). Note that
        preload=False will be effective only if the data is stored in a
        separate binary file.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    raw : Instance of RawSet
        A Raw object containing EEGLAB .set data.

    Notes
    -----
    .. versionadded:: 0.11.0

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawSet(fname=fname, montage=montage, eog=eog, preload=preload,
                  verbose=verbose)


class RawSet(_BaseRaw):
    """Raw object from EEGLAB .set file.

    Parameters
    ----------
    fname : str
        Path to the .set file.
    montage : str | None | instance of montage
        Path or instance of montage containing electrode positions.
        If None, sensor locations are (0,0,0). See the documentation of
        :func:`mne.channels.read_montage` for more information.
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

    Notes
    -----
    .. versionadded:: 0.11.0

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    def __init__(self, fname, montage, eog=None, preload=False, verbose=None):
        """Read EEGLAB .set file.
        """
        from scipy import io
        basedir = op.dirname(fname)
        eeg = io.loadmat(fname, struct_as_record=False, squeeze_me=True)['EEG']

        if not isinstance(eeg.data, basestring) and not preload:
            warnings.warn('Data will be preloaded. preload=False is not '
                          'supported when the data is stored in the .set file')
        if eeg.trials != 1:
            raise ValueError('The number of trials is %d. It must be 1 for raw'
                             ' files' % eeg.trials)

        last_samps = [eeg.pnts - 1]
        info = _get_info(eeg, montage, eog)

        # read the data
        if isinstance(eeg.data, basestring):
            data_fname = op.join(basedir, eeg.data)
            logger.info('Reading %s' % data_fname)

            super(RawSet, self).__init__(
                info, preload, filenames=[data_fname], last_samps=last_samps,
                orig_format='double', verbose=verbose)
        else:
            data = eeg.data.reshape(eeg.nbchan, -1, order='F')
            data = data.astype(np.double)
            super(RawSet, self).__init__(
                info, data, filenames=[fname], last_samps=last_samps,
                orig_format='double', verbose=verbose)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data"""
        n_bytes = 4
        nchan = self.info['nchan']
        data_offset = self.info['nchan'] * start * n_bytes
        data_left = (stop - start) * nchan
        # Read up to 100 MB of data at a time.
        n_blocks = 100000000 // n_bytes
        blk_size = min(data_left, (n_blocks // nchan) * nchan)

        with open(self._filenames[fi], 'rb', buffering=0) as fid:
            fid.seek(data_offset)
            # extract data in chunks
            for blk_start in np.arange(0, data_left, blk_size) // nchan:
                blk_size = min(blk_size, data_left - blk_start * nchan)
                block = np.fromfile(fid,
                                    dtype=np.float32, count=blk_size)
                block = block.reshape(nchan, -1, order='F')
                blk_stop = blk_start + block.shape[1]
                data_view = data[:, blk_start:blk_stop]
                _mult_cal_one(data_view, block, idx, cals, mult)
        return data
