"""Conversion tool from CTF to FIF."""

# Authors: Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#          Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import os
import os.path as op

import numpy as np

from .._digitization import _format_dig_points
from ...utils import (verbose, logger, _clean_names, fill_doc, _check_option,
                      _validate_type)

from ..base import BaseRaw
from ..utils import _mult_cal_one, _blk_read_lims

from .res4 import _read_res4, _make_ctf_name
from .hc import _read_hc
from .eeg import _read_eeg, _read_pos
from .trans import _make_ctf_coord_trans_set
from .info import _compose_meas_info, _read_bad_chans, _annotate_bad_segments
from .constants import CTF
from .markers import _read_annotations_ctf_call


@fill_doc
def read_raw_ctf(directory, system_clock='truncate', preload=False,
                 clean_names=False, verbose=None):
    """Raw object from CTF directory.

    Parameters
    ----------
    directory : str
        Path to the CTF data (ending in ``'.ds'``).
    system_clock : str
        How to treat the system clock. Use "truncate" (default) to truncate
        the data file when the system clock drops to zero, and use "ignore"
        to ignore the system clock (e.g., if head positions are measured
        multiple times during a recording).
    %(preload)s
    clean_names : bool, optional
        If True main channel names and compensation channel names will
        be cleaned from CTF suffixes. The default is False.
    %(verbose)s

    Returns
    -------
    raw : instance of RawCTF
        The raw data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.

    Notes
    -----
    .. versionadded:: 0.11
    """
    return RawCTF(directory, system_clock, preload=preload,
                  clean_names=clean_names, verbose=verbose)


@fill_doc
class RawCTF(BaseRaw):
    """Raw object from CTF directory.

    Parameters
    ----------
    directory : str
        Path to the CTF data (ending in ``'.ds'``).
    system_clock : str
        How to treat the system clock. Use "truncate" (default) to truncate
        the data file when the system clock drops to zero, and use "ignore"
        to ignore the system clock (e.g., if head positions are measured
        multiple times during a recording).
    %(preload)s
    clean_names : bool, optional
        If True main channel names and compensation channel names will
        be cleaned from CTF suffixes. The default is False.
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    @verbose
    def __init__(self, directory, system_clock='truncate', preload=False,
                 verbose=None, clean_names=False):  # noqa: D102
        # adapted from mne_ctf2fiff.c
        _validate_type(directory, 'path-like', 'directory')
        directory = str(directory)
        if not directory.endswith('.ds'):
            raise TypeError('directory must be a directory ending with ".ds", '
                            f'got {directory}')
        if not op.isdir(directory):
            raise ValueError('directory does not exist: "%s"' % directory)
        _check_option('system_clock', system_clock, ['ignore', 'truncate'])
        logger.info('ds directory : %s' % directory)
        res4 = _read_res4(directory)  # Read the magical res4 file
        coils = _read_hc(directory)  # Read the coil locations
        eeg = _read_eeg(directory)  # Read the EEG electrode loc info

        # Investigate the coil location data to get the coordinate trans
        coord_trans = _make_ctf_coord_trans_set(res4, coils)

        digs = _read_pos(directory, coord_trans)

        # Compose a structure which makes fiff writing a piece of cake
        info = _compose_meas_info(res4, coils, coord_trans, eeg)
        info['dig'] += digs
        info['dig'] = _format_dig_points(info['dig'])
        info['bads'] += _read_bad_chans(directory, info)

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
            sample_info = _get_sample_info(meg4_name, res4, system_clock)
            if sample_info['n_samp'] == 0:
                break
            if len(fnames) == 0:
                buffer_size_sec = sample_info['block_size'] / info['sfreq']
            else:
                buffer_size_sec = 1.
            fnames.append(meg4_name)
            last_samps.append(sample_info['n_samp'] - 1)
            raw_extras.append(sample_info)
            first_samps = [0] * len(last_samps)
        super(RawCTF, self).__init__(
            info, preload, first_samps=first_samps,
            last_samps=last_samps, filenames=fnames,
            raw_extras=raw_extras, orig_format='int',
            buffer_size_sec=buffer_size_sec, verbose=verbose)

        # Add bad segments as Annotations (correct for start time)
        start_time = -res4['pre_trig_pts'] / float(info['sfreq'])
        annot = _annotate_bad_segments(directory, start_time,
                                       info['meas_date'])
        marker_annot = _read_annotations_ctf_call(
            directory=directory,
            total_offset=(res4['pre_trig_pts'] / res4['sfreq']),
            trial_duration=(res4['nsamp'] / res4['sfreq']),
            meas_date=info['meas_date']
        )
        annot = marker_annot if annot is None else annot + marker_annot
        self.set_annotations(annot)

        if clean_names:
            self._clean_names()

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        si = self._raw_extras[fi]
        offset = 0
        trial_start_idx, r_lims, d_lims = _blk_read_lims(start, stop,
                                                         int(si['block_size']))
        with open(self._filenames[fi], 'rb') as fid:
            for bi in range(len(r_lims)):
                samp_offset = (bi + trial_start_idx) * si['res4_nsamp']
                n_read = min(si['n_samp_tot'] - samp_offset, si['block_size'])
                # read the chunk of data
                pos = CTF.HEADER_SIZE
                pos += samp_offset * si['n_chan'] * 4
                fid.seek(pos, 0)
                this_data = np.fromfile(fid, '>i4',
                                        count=si['n_chan'] * n_read)
                this_data.shape = (si['n_chan'], n_read)
                this_data = this_data[:, r_lims[bi, 0]:r_lims[bi, 1]]
                data_view = data[:, d_lims[bi, 0]:d_lims[bi, 1]]
                _mult_cal_one(data_view, this_data, idx, cals, mult)
                offset += n_read

    def _clean_names(self):
        """Clean up CTF suffixes from channel names."""
        mapping = dict(zip(self.ch_names, _clean_names(self.ch_names)))

        self.rename_channels(mapping)

        for comp in self.info['comps']:
            for key in ('row_names', 'col_names'):
                comp['data'][key] = _clean_names(comp['data'][key])


def _get_sample_info(fname, res4, system_clock):
    """Determine the number of valid samples."""
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
        elif system_clock == 'ignore':
            logger.info('    System clock channel is available, but ignored.')
        else:  # use it
            logger.info('    System clock channel is available, checking '
                        'which samples are valid.')
            for t in range(n_trial):
                # Skip to the correct trial
                samp_offset = t * res4['nsamp']
                offset = CTF.HEADER_SIZE + (samp_offset * res4['nchan'] +
                                            (clock_ch * res4['nsamp'])) * 4
                fid.seek(offset, 0)
                this_data = np.fromfile(fid, '>i4', res4['nsamp'])
                if len(this_data) != res4['nsamp']:
                    raise RuntimeError('Cannot read data for trial %d'
                                       % (t + 1))
                end = np.where(this_data == 0)[0]
                if len(end) > 0:
                    n_samp = samp_offset + end[0]
                    break
    if n_samp < res4['nsamp']:
        n_trial = 1
        logger.info('    %d x %d = %d samples from %d chs'
                    % (n_trial, n_samp, n_samp, res4['nchan']))
    else:
        n_trial = n_samp // res4['nsamp']
        n_omit = n_samp_tot - n_samp
        logger.info('    %d x %d = %d samples from %d chs'
                    % (n_trial, res4['nsamp'], n_samp, res4['nchan']))
        if n_omit != 0:
            logger.info('    %d samples omitted at the end' % n_omit)

    return dict(n_samp=n_samp, n_samp_tot=n_samp_tot, block_size=res4['nsamp'],
                res4_nsamp=res4['nsamp'], n_chan=res4['nchan'])
