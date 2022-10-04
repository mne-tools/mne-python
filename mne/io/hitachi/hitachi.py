# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

import datetime as dt
import re

import numpy as np

from ..base import BaseRaw
from ..constants import FIFF
from ..meas_info import create_info, _merge_info
from ..nirx.nirx import _read_csv_rows_cols
from ..utils import _mult_cal_one
from ...utils import (logger, verbose, fill_doc, warn, _check_fname,
                      _check_option)


@fill_doc
def read_raw_hitachi(fname, preload=False, verbose=None):
    """Reader for a Hitachi fNIRS recording.

    Parameters
    ----------
    %(hitachi_fname)s
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawHitachi
        A Raw object containing Hitachi data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.

    Notes
    -----
    %(hitachi_notes)s
    """
    return RawHitachi(fname, preload, verbose=verbose)


def _check_bad(cond, msg):
    if cond:
        raise RuntimeError(f'Could not parse file: {msg}')


@fill_doc
class RawHitachi(BaseRaw):
    """Raw object from a Hitachi fNIRS file.

    Parameters
    ----------
    %(hitachi_fname)s
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.

    Notes
    -----
    %(hitachi_notes)s
    """

    @verbose
    def __init__(self, fname, preload=False, *, verbose=None):
        if not isinstance(fname, (list, tuple)):
            fname = [fname]
        fname = list(fname)  # our own list that we can modify
        for fi, this_fname in enumerate(fname):
            fname[fi] = _check_fname(this_fname, 'read', True, f'fname[{fi}]')
        infos = list()
        probes = list()
        last_samps = list()
        S_offset = D_offset = 0
        ignore_names = ['Time']
        for this_fname in fname:
            info, extra, last_samp, offsets = self._get_hitachi_info(
                this_fname, S_offset, D_offset, ignore_names)
            ignore_names = list(set(ignore_names + info['ch_names']))
            S_offset += offsets[0]
            D_offset += offsets[1]
            infos.append(info)
            probes.append(extra)
            last_samps.append(last_samp)
        # combine infos
        if len(fname) > 1:
            info = _merge_info(infos)
        else:
            info = infos[0]
        if len(set(last_samps)) != 1:
            raise RuntimeError('All files must have the same number of '
                               'samples, got: {last_samps}')
        last_samps = [last_samps[0]]
        raw_extras = [dict(probes=probes)]
        # One representative filename is good enough here
        # (additional filenames indicate temporal concat, not ch concat)
        filenames = [fname[0]]
        super().__init__(
            info, preload, filenames=filenames, last_samps=last_samps,
            raw_extras=raw_extras, verbose=verbose)

    # This could be a function, but for the sake of indentation, let's make it
    # a method instead
    def _get_hitachi_info(self, fname, S_offset, D_offset, ignore_names):
        logger.info('Loading %s' % fname)
        raw_extra = dict(fname=fname)
        info_extra = dict()
        subject_info = dict()
        ch_wavelengths = dict()
        fnirs_wavelengths = [None, None]
        meas_date = age = ch_names = sfreq = None
        with open(fname, 'rb') as fid:
            lines = fid.read()
        lines = lines.decode('latin-1').rstrip('\r\n')
        oldlen = len(lines)
        assert len(lines) == oldlen
        bounds = [0]
        end = '\n' if '\n' in lines else '\r'
        bounds.extend(a.end() for a in re.finditer(end, lines))
        bounds.append(len(lines))
        lines = lines.split(end)
        assert len(bounds) == len(lines) + 1
        line = lines[0].rstrip(',\r\n')
        _check_bad(line != 'Header', 'no header found')
        li = 0
        mode = None
        for li, line in enumerate(lines[1:], 1):
            # Newer format has some blank lines
            if len(line) == 0:
                continue
            parts = line.rstrip(',\r\n').split(',')
            if len(parts) == 0:  # some header lines are blank
                continue
            kind, parts = parts[0], parts[1:]
            if len(parts) == 0:
                parts = ['']  # some fields (e.g., Comment) meaningfully blank
            if kind == 'File Version':
                logger.info(f'Reading Hitachi fNIRS file version {parts[0]}')
            elif kind == 'AnalyzeMode':
                _check_bad(
                    parts != ['Continuous'], f'not continuous data ({parts})')
            elif kind == 'Sampling Period[s]':
                sfreq = 1 / float(parts[0])
            elif kind == 'Exception':
                raise NotImplementedError(kind)
            elif kind == 'Comment':
                info_extra['description'] = parts[0]
            elif kind == 'ID':
                subject_info['his_id'] = parts[0]
            elif kind == 'Name':
                if len(parts):
                    name = parts[0].split(' ')
                    if len(name):
                        subject_info['first_name'] = name[0]
                        subject_info['last_name'] = ' '.join(name[1:])
            elif kind == 'Age':
                age = int(parts[0].rstrip('y'))
            elif kind == 'Mode':
                mode = parts[0]
            elif kind in ('HPF[Hz]', 'LPF[Hz]'):
                try:
                    freq = float(parts[0])
                except ValueError:
                    pass
                else:
                    info_extra[{'HPF[Hz]': 'highpass',
                                'LPF[Hz]': 'lowpass'}[kind]] = freq
            elif kind == 'Date':
                # 5/17/04 5:14
                try:
                    mdy, HM = parts[0].split(' ')
                    H, M = HM.split(':')
                    if len(H) == 1:
                        H = f'0{H}'
                    mdyHM = ' '.join([mdy, ':'.join([H, M])])
                    for fmt in ('%m/%d/%y %H:%M', '%Y/%m/%d %H:%M'):
                        try:
                            meas_date = dt.datetime.strptime(mdyHM, fmt)
                        except Exception:
                            pass
                        else:
                            break
                    else:
                        raise RuntimeError  # unknown format
                except Exception:
                    warn('Extraction of measurement date failed. '
                         'Please report this as a github issue. '
                         'The date is being set to January 1st, 2000, '
                         f'instead of {repr(parts[0])}')
            elif kind == 'Sex':
                try:
                    subject_info['sex'] = dict(
                        female=FIFF.FIFFV_SUBJ_SEX_FEMALE,
                        male=FIFF.FIFFV_SUBJ_SEX_MALE)[parts[0].lower()]
                except KeyError:
                    pass
            elif kind == 'Wave[nm]':
                fnirs_wavelengths[:] = [int(part) for part in parts]
            elif kind == 'Wave Length':
                ch_regex = re.compile(r'^(.*)\(([0-9\.]+)\)$')
                for ent in parts:
                    _, v = ch_regex.match(ent).groups()
                    ch_wavelengths[ent] = float(v)
            elif kind == 'Data':
                break
        fnirs_wavelengths = np.array(fnirs_wavelengths, int)
        assert len(fnirs_wavelengths) == 2
        ch_names = lines[li + 1].rstrip(',\r\n').split(',')
        # cull to correct ones
        raw_extra['keep_mask'] = ~np.in1d(ch_names, list(ignore_names))
        for ci, ch_name in enumerate(ch_names):
            if re.match('Probe[0-9]+', ch_name):
                raw_extra['keep_mask'][ci] = False
        # set types
        ch_names = [ch_name for ci, ch_name in enumerate(ch_names)
                    if raw_extra['keep_mask'][ci]]
        ch_types = ['fnirs_cw_amplitude' if ch_name.startswith('CH')
                    else 'stim'
                    for ch_name in ch_names]
        # get locations
        nirs_names = [ch_name for ch_name, ch_type in zip(ch_names, ch_types)
                      if ch_type == 'fnirs_cw_amplitude']
        n_nirs = len(nirs_names)
        assert n_nirs % 2 == 0
        names = {
            '3x3': 'ETG-100',
            '3x5': 'ETG-7000',
            '4x4': 'ETG-7000',
            '3x11': 'ETG-4000',
        }
        _check_option('Hitachi mode', mode, sorted(names))
        n_row, n_col = [int(x) for x in mode.split('x')]
        logger.info(f'Constructing pairing matrix for {names[mode]} ({mode})')
        pairs = _compute_pairs(n_row, n_col, n=1 + (mode == '3x3'))
        assert n_nirs == len(pairs) * 2
        locs = np.zeros((len(ch_names), 12))
        locs[:, :9] = np.nan
        idxs = np.where(np.array(ch_types, 'U') == 'fnirs_cw_amplitude')[0]
        for ii, idx in enumerate(idxs):
            ch_name = ch_names[idx]
            # Use the actual/accurate wavelength in loc
            acc_freq = ch_wavelengths[ch_name]
            locs[idx][9] = acc_freq
            # Rename channel based on standard naming scheme, using the
            # nominal wavelength
            sidx, didx = pairs[ii // 2]
            nom_freq = fnirs_wavelengths[np.argmin(np.abs(
                acc_freq - fnirs_wavelengths))]
            ch_names[idx] = (
                f'S{S_offset + sidx + 1}_'
                f'D{D_offset + didx + 1} '
                f'{nom_freq}'
            )
        offsets = np.array(pairs, int).max(axis=0) + 1

        # figure out bounds
        bounds = raw_extra['bounds'] = bounds[li + 2:]
        last_samp = len(bounds) - 2

        if age is not None and meas_date is not None:
            subject_info['birthday'] = (meas_date.year - age,
                                        meas_date.month,
                                        meas_date.day)
        if meas_date is None:
            meas_date = dt.datetime(2000, 1, 1, 0, 0, 0)
        meas_date = meas_date.replace(tzinfo=dt.timezone.utc)
        if subject_info:
            info_extra['subject_info'] = subject_info

        # Create mne structure
        info = create_info(ch_names, sfreq, ch_types=ch_types)
        with info._unlock():
            info.update(info_extra)
            info['meas_date'] = meas_date
            for li, loc in enumerate(locs):
                info['chs'][li]['loc'][:] = loc
        return info, raw_extra, last_samp, offsets

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a segment of data from a file."""
        this_data = list()
        for this_probe in self._raw_extras[fi]['probes']:
            this_data.append(_read_csv_rows_cols(
                this_probe['fname'],
                start, stop, this_probe['keep_mask'],
                this_probe['bounds'], sep=',',
                replace=lambda x:
                    x.replace('\r', '\n')
                    .replace('\n\n', '\n')
                    .replace('\n', ',')
                    .replace(':', '')).T)
        this_data = np.concatenate(this_data, axis=0)
        _mult_cal_one(data, this_data, idx, cals, mult)
        return data


def _compute_pairs(n_rows, n_cols, n=1):
    n_tot = n_rows * n_cols
    sd_idx = (np.arange(n_tot) // 2).reshape(n_rows, n_cols)
    d_bool = np.empty((n_rows, n_cols), bool)
    for ri in range(n_rows):
        d_bool[ri] = np.arange(ri, ri + n_cols) % 2
    pairs = list()
    for ri in range(n_rows):
        # First iterate over connections within the row
        for ci in range(n_cols - 1):
            pair = (sd_idx[ri, ci], sd_idx[ri, ci + 1])
            if d_bool[ri, ci]:  # reverse
                pair = pair[::-1]
            pairs.append(pair)
        # Next iterate over row-row connections, if applicable
        if ri >= n_rows - 1:
            continue
        for ci in range(n_cols):
            pair = (sd_idx[ri, ci], sd_idx[ri + 1, ci])
            if d_bool[ri, ci]:
                pair = pair[::-1]
            pairs.append(pair)
    if n > 1:
        assert n == 2  # only one supported for now
        pairs = np.array(pairs, int)
        second = pairs + pairs.max(axis=0) + 1
        pairs = np.r_[pairs, second]
        pairs = tuple(tuple(row) for row in pairs)
    return tuple(pairs)
