"""Conversion tool from CTF to FIF
"""

# Author: Eric Larson <larson.eric.d<gmail.com>
#
# License: BSD (3-clause)

import os
from os import path as op

import numpy as np

from ...utils import verbose, logger
from ..base import _BaseRaw, _mult_cal_one
from ...externals.six import string_types

from .res4 import _read_res4, _make_ctf_name
from .hc import _read_hc
from .eeg import _read_eeg
from .trans import _make_ctf_coord_trans_set
from .info import _compose_meas_info
from .constants import CTF


def read_raw_ctf(directory, preload=False, verbose=None):
    return RawCTF(directory, preload, verbose)


class RawCTF(_BaseRaw):
    """Raw object from CTF directory

    Parameters
    ----------
    directory : str
        Path to the KIT data (ending in ``'.ds'``).
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
    def __init__(self, directory, preload=False, verbose=None):
        # adapted from mne_ctf2fiff.c
        if not isinstance(directory, string_types) or \
                not directory.endswith('.ds'):
            raise TypeError('directory must be a directory ending with ".ds"')
        if not op.isdir(directory):
            raise ValueError('directory does not exist: "%s"' % directory)
        logger.info('ds directory : %s' % directory)
        res4 = _read_res4(directory)  # Read the magical res4 file
        coils = _read_hc(directory)  # Read the coil locations
        eeg = _read_eeg(directory)  # Read the EEG electrode loc info
        # Investigate the coil location data to get the coordinate trans
        coord_trans = _make_ctf_coord_trans_set(res4, coils)
        # Compose a structure which makes fiff writing a piece of cake
        info = _compose_meas_info(res4, coils, coord_trans, eeg)
        # Determine how our data is distributed across files
        fnames = list()
        last_samps = list()
        raw_extras = list()
        while(True):
            suffix = 'meg4' if len(fnames) == 0 else ('%d_meg4' % len(fnames))
            meg4_name = _make_ctf_name(directory, suffix, raise_error=False)
            if meg4_name is None:
                break
            # check how much data is in the file
            sample_info = _get_sample_info(meg4_name, res4)
            if sample_info['n_samp'] == 0:
                break
            fnames.append(meg4_name)
            last_samps.append(sample_info['n_samp'] - 1)
            raw_extras.append(sample_info)
        super(RawCTF, self).__init__(
            info, preload, last_samps=last_samps, filenames=fnames,
            raw_extras=raw_extras, orig_format='int', verbose=verbose)

    @verbose
    def _read_segment_file(self, data, idx, offset, fi, start, stop,
                           cals, mult):
        """Read a chunk of raw data"""
        si = self._raw_extras[fi]
        with open(self._filenames[fi], 'rb') as fid:
            read_offsets = np.arange(start, stop + 1, si['block_size'])
            for t, read_offset in enumerate(read_offsets):
                n_read = min(stop + 1 - read_offset, si['block_size'])
                data_view = data[:, offset:offset + n_read]
                # read the chunk of data
                pos = CTF.HEADER_SIZE
                if si['n_trial'] == 1:
                    pos += read_offset * 4
                else:
                    pos += t * (si['res4_nsamp'] * si['n_chan']) * 4
                fid.seek(pos, 0)
                one = np.fromstring(fid.read(si['n_chan'] * n_read * 4), '>i4')
                if len(one) != si['n_chan'] * n_read:
                    raise RuntimeError('Cannot read data')
                one.shape = (si['n_chan'], n_read)
                _mult_cal_one(data_view, one, idx, fi, cals, mult)
                offset += n_read


def _get_sample_info(fname, res4):
    """Helper to determine the number of valid samples"""
    logger.info('Finding samples for %s: ' % (fname,))
    clock_ch = -1
    data = None
    for k, ch in enumerate(res4['chs']):
        if ch['name'] == CTF.SYSTEM_CLOCK_CH:
            clock_ch = k
            break
    with open(fname, 'rb') as fid:
        fid.seek(0, os.SEEK_END)
        st_size = fid.tell()
        fid.seek(0, 0)
        if (st_size - CTF.HEADER_SIZE) % (4 * res4['nchan']) != 0:
            raise RuntimeError('The number of samples is not an even multiple '
                               'of the number of channels')
        if (st_size - CTF.HEADER_SIZE) % (4 * res4['nsamp'] *
                                          res4['nchan']) != 0:
            raise RuntimeError('The number of samples is not an even multiple '
                               'of the trial size')
        n_samp_tot = (st_size - CTF.HEADER_SIZE) // (4 * res4['nchan'])
        n_trial = n_samp_tot // res4['nsamp']
        if clock_ch < 0:
            logger.info('    System clock channel is not available, assuming '
                        'all samples to be valid.')
            n_samp = n_samp_tot
        else:
            logger.info('    System clock channel is available, checking '
                        'which samples are valid.')
            data = np.empty(n_samp_tot, np.int32)
            t = 0
            for t in range(n_trial):
                this_sl = slice(t * res4['nsamp'], (t + 1) * res4['nsamp'])
                # Skip to the correct trial
                offset = CTF.HEADER_SIZE + (t * (res4['nsamp'] *
                                                 res4['nchan']) +
                                            (clock_ch * res4['nsamp'])) * 4
                fid.seek(offset, 0)
                this_data = np.fromstring(fid.read(4 * res4['nsamp']), '>i4')
                if len(this_data) != res4['nsamp']:
                    raise RuntimeError('Cannot read data for trial %d'
                                       % (t + 1))
                data[this_sl] = this_data
            end = np.where(data == 0)[0]
            n_samp = end[0] if len(end) > 0 else n_samp_tot
    if n_samp < res4['nsamp']:
        n_trial = 1
        logger.info('    %d x %d = %d samples from %d chs'
                    % (n_trial, n_samp, n_samp, res4['nchan']))
    else:
        n_trial = n_samp // res4['nsamp']
        n_omit = n_samp % res4['nsamp']
        n_samp = n_trial * res4['nsamp']
        logger.info('    %d x %d = %d samples from %d chs'
                    % (n_trial, res4['nsamp'], n_samp, res4['nchan']))
        if n_omit != 0:
            logger.info('    %d samples omitted at the end' % n_omit)
    if n_trial == 1:
        block_size = 2000
    else:
        block_size = res4['nsamp']
    return dict(n_samp=n_samp, n_samp_tot=n_samp_tot, block_size=block_size,
                n_trial=n_trial, res4_nsamp=res4['nsamp'],
                n_chan=res4['nchan'])
