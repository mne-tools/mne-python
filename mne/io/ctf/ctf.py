"""Conversion tool from CTF to FIF
"""

# Author: Eric Larson <larson.eric.d<gmail.com>
#
# License: BSD (3-clause)

import os
from os import path as op

import numpy as np

from ...utils import verbose, logger
from ...externals.six import string_types

from ..base import _BaseRaw, _mult_cal_one, _block_read_lims, _triage_reads

from .res4 import _read_res4, _make_ctf_name
from .hc import _read_hc
from .eeg import _read_eeg
from .trans import _make_ctf_coord_trans_set
from .info import _compose_meas_info
from .constants import CTF


def read_raw_ctf(directory, preload=False, verbose=None):
    """Raw object from CTF directory

    Parameters
    ----------
    directory : str
        Path to the CTF data (ending in ``'.ds'``).
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
    raw : instance of RawCTF
        The raw data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawCTF(directory, preload, verbose)


class RawCTF(_BaseRaw):
    """Raw object from CTF directory

    Parameters
    ----------
    directory : str
        Path to the CTF data (ending in ``'.ds'``).
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
            if len(fnames) == 0:
                info['buffer_size_sec'] = \
                    sample_info['block_size'] / info['sfreq']
                info['filename'] = directory
            fnames.append(meg4_name)
            last_samps.append(sample_info['n_samp'] - 1)
            raw_extras.append(sample_info)
        super(RawCTF, self).__init__(
            info, preload, last_samps=last_samps, filenames=fnames,
            raw_extras=raw_extras, orig_format='int', verbose=verbose)

    @verbose
    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data"""
        si = self._raw_extras[fi]
        idx, read_chs = _triage_reads(mult, idx, si['n_chan'])
        block_start_idx, r_lims, d_lims = _block_read_lims(
            start, stop, si['block_size'])
        with open(self._filenames[fi], 'rb') as fid:
            for bi in range(len(r_lims)):
                n_read = r_lims[bi, 1] - r_lims[bi, 0]
                block_offset = (CTF.HEADER_SIZE +
                                (bi + block_start_idx) * si['block_size'] *
                                si['n_chan'] * 4)
                this_data = np.empty((len(read_chs), n_read), data.dtype)
                for ii, ci in enumerate(read_chs):
                    # read the proper chunk of data:
                    #     move to the start of the block,
                    #     move to the start of the channel within the block,
                    #     move to the first used sample of data in the channel
                    pos = (block_offset +
                           (ci * si['block_size'] +
                            r_lims[bi, 0]) * 4)
                    fid.seek(pos, 0)
                    this_data[ii] = np.fromfile(fid, '>i4', n_read)
                data_view = data[:, d_lims[bi, 0]:d_lims[bi, 1]]
                _mult_cal_one(data_view, this_data, idx, cals, mult)


def _get_sample_info(fname, res4):
    """Helper to determine the number of valid samples"""
    logger.info('Finding samples for %s: ' % (fname,))
    if CTF.SYSTEM_CLOCK_CH in res4['ch_names']:
        clock_ch = res4['ch_names'].index(CTF.SYSTEM_CLOCK_CH)
    else:
        clock_ch = None
    for k, ch in enumerate(res4['chs']):
        if ch['ch_name'] == CTF.SYSTEM_CLOCK_CH:
            clock_ch = k
            break
    with open(fname, 'rb') as fid:
        fid.seek(0, os.SEEK_END)
        st_size = fid.tell()
        fid.seek(0, 0)
        if (st_size - CTF.HEADER_SIZE) % (4 * res4['nsamp'] *
                                          res4['nchan']) != 0:
            raise RuntimeError('The number of samples is not an even multiple '
                               'of the trial size')
        n_samp_tot = (st_size - CTF.HEADER_SIZE) // (4 * res4['nchan'])
        n_trial = n_samp_tot // res4['nsamp']
        n_samp = n_samp_tot
        if clock_ch is None:
            logger.info('    System clock channel is not available, assuming '
                        'all samples to be valid.')
        else:
            logger.info('    System clock channel is available, checking '
                        'which samples are valid.')
            for t in range(n_trial):
                # Skip to the correct trial
                offset = CTF.HEADER_SIZE + (t * (res4['nsamp'] *
                                                 res4['nchan']) +
                                            (clock_ch * res4['nsamp'])) * 4
                fid.seek(offset, 0)
                this_data = np.fromstring(fid.read(4 * res4['nsamp']), '>i4')
                if len(this_data) != res4['nsamp']:
                    raise RuntimeError('Cannot read data for trial %d'
                                       % (t + 1))
                end = np.where(this_data == 0)[0]
                if len(end) > 0:
                    n_samp = t * res4['nsamp'] + end[0]
                    break
    if n_samp < res4['nsamp']:
        n_trial = 1
        n_omit = 0
    else:
        n_trial = n_samp // res4['nsamp']
        n_omit = n_samp % res4['nsamp']
        n_samp = n_trial * res4['nsamp']
    logger.info('    %d x %d = %d samples from %d chs'
                % (n_trial, res4['nsamp'], n_samp, res4['nchan']))
    if n_omit != 0:
        logger.info('    %d samples omitted at the end' % n_omit)
    return dict(n_samp=n_samp, n_samp_tot=n_samp_tot, n_trial=n_trial,
                block_size=int(res4['nsamp']), n_chan=res4['nchan'])
