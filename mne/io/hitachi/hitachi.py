# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

from collections import Counter
import datetime as dt
import re

import numpy as np

from ..base import BaseRaw
from ..constants import FIFF
from ..meas_info import create_info
from ..nirx.nirx import _read_csv_rows_cols
from ..utils import _mult_cal_one
from ...utils import logger, verbose, fill_doc, warn, _check_fname


@fill_doc
def read_raw_hitachi(fname, preload=False, verbose=None):
    """Reader for a NIRX fNIRS recording.

    Parameters
    ----------
    fname : str
        Path to the Hitachi CSV file.
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
    .. versionadded:: 0.24
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
    fname : str
        Path to the Hitachi CSV file.
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.

    Notes
    -----
    Hitachi does not encode their channel positions, so you will need to
    create a suitable mapping using :func:`mne.channels.make_standard_montage`
    or :func:`mne.channels.make_dig_montage` like:

    >>> mon = mne.channels.make_standard_montage('standard_1020')
    >>> need = 'S1 D1 S2 D2 S3 D3 S4 D4 S5 D5 S6 D6 S7 D7 S8'.split()
    >>> have = 'F7 F5 FC5 F3 FC3 FT7 T7 C5 C3 TP7 CP5 CP3 P7 P5 P3'.split()
    >>> mon.rename_channels(dict(zip(have, need)))
    >>> raw.set_montage(mon)  # doctest: +SKIP

    .. versionadded:: 0.24
    """

    @verbose
    def __init__(self, fname, preload=False, *, verbose=None):
        fname = _check_fname(fname, 'read', True, 'fname')
        logger.info('Loading %s' % fname)

        raw_extra = dict(fname=fname)
        info_extra = dict()
        subject_info = dict()
        ch_wavelengths = dict()
        fnirs_wavelengths = [None, None]
        meas_date = age = ch_names = sfreq = None
        with open(fname, 'rb') as fid:
            lines = fid.read()
        lines = lines.decode('latin-1').rstrip('\r')
        oldlen = len(lines)
        assert len(lines) == oldlen
        bounds = [0]
        bounds.extend(a.end() for a in re.finditer('\r', lines))
        bounds.append(len(lines))
        lines = lines.split('\r')
        assert len(bounds) == len(lines) + 1
        line = lines[0].rstrip(',')
        _check_bad(line != 'Header', 'no header found')
        li = 0
        for li, line in enumerate(lines[1:], 1):
            _check_bad(len(line) == 0, 'no data found')
            parts = line.rstrip(',').split(',')
            if len(parts) == 0:  # some header lines are blank
                continue
            kind, parts = parts[0], parts[1:]
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
                name = parts[0].split(' ')
                if len(name):
                    subject_info['first_name'] = name[0]
                    subject_info['last_name'] = ' '.join(name[1:])
            elif kind == 'Age':
                age = int(parts[0].rstrip('y'))
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
                    meas_date = dt.datetime.strptime(mdyHM, '%m/%d/%y %H:%M')
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
                regex = re.compile(r'^(.*)\(([0-9\.]+)\)$')
                for ent in parts:
                    _, v = regex.match(ent).groups()
                    ch_wavelengths[ent] = float(v)
            elif kind == 'Data':
                break
        fnirs_wavelengths = np.array(fnirs_wavelengths, int)
        assert len(fnirs_wavelengths) == 2
        ch_names = lines[li + 1].rstrip().rstrip(',').split(',')
        # cull to correct ones
        raw_extra['keep_mask'] = ~np.in1d(ch_names, ['Probe1', 'Time'])
        # set types
        ch_names = [ch_name for ci, ch_name in enumerate(ch_names)
                    if raw_extra['keep_mask'][ci]]
        ch_types = ['fnirs_cw_amplitude' if ch_name.startswith('CH')
                    else 'stim'
                    for ch_name in ch_names]
        # get locations
        n_nirs = Counter(ch_types).get('fnirs_cw_amplitude', 0)
        if n_nirs != 44:
            raise RuntimeError('Only Hitachi 3x5 with 44 total channels '
                               f'supported, got {n_nirs}')
        pairs = (
            (0, 0), (1, 0), (1, 1), (2, 1), (0, 2),
            (3, 0), (1, 3), (4, 1), (2, 4), (3, 2),
            (3, 3), (4, 3), (4, 4), (5, 2), (3, 5),
            (6, 3), (4, 6), (7, 4), (5, 5), (6, 5),
            (6, 6), (7, 6))
        assert len(pairs) == n_nirs // 2
        locs = np.zeros((len(ch_names), 12))
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
            ch_names[idx] = f'S{sidx + 1}_D{didx + 1} {nom_freq}'

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
        info.update(info_extra)
        info['meas_date'] = meas_date
        for li, loc in enumerate(locs):
            info['chs'][li]['loc'][:] = loc

        super().__init__(
            info, preload, filenames=[fname], last_samps=[last_samp],
            raw_extras=[raw_extra], verbose=verbose)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a segment of data from a file."""
        this_data = _read_csv_rows_cols(
            self._raw_extras[fi]['fname'],
            start, stop, self._raw_extras[fi]['keep_mask'],
            self._raw_extras[fi]['bounds'], sep=',',
            replace=lambda x: x.replace('\r', ',').replace(':', '')).T
        _mult_cal_one(data, this_data, idx, cals, mult)
        return data
