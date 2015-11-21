"""Read .res4 files
"""

# Author: Eric Larson <larson.eric.d<gmail.com>
#
# License: BSD (3-clause)

import os.path as op

import numpy as np

from ...utils import logger
from .constants import CTF


def _make_ctf_name(directory, extra, raise_error=True):
    """Helper to make a CTF name"""
    fname = op.join(directory, op.basename(directory)[:-3] + '.' + extra)
    if not op.isfile(fname):
        if raise_error:
            raise IOError('Standard file %s not found' % fname)
        else:
            return None
    return fname


def _read_double(fid):
    """Read a double"""
    return np.fromstring(fid.read(8), '>f8')[0]


def _read_string(fid, n_bytes, decode=True):
    """Read string"""
    s0 = fid.read(n_bytes)
    s = s0.split('\x00')[0]
    return s.decode('utf-8') if decode else s


def _read_ustring(fid, n_bytes):
    """Read unsigned character string"""
    return np.fromstring(fid.read(n_bytes), '>B')


def _read_int2(fid):
    """Read int from short"""
    return np.fromstring(fid.read(2), '>i2')[0]


def _read_int(fid):
    """Read a 32-bit integer"""
    return np.fromstring(fid.read(4), '>i4')[0]


def _move_to_next(fid, byte=8):
    """Move to next byte boundary"""
    now = fid.tell()
    if now % byte != 0:
        now = now - (now % byte) + byte
        fid.seek(now, 0)


def _read_filter(fid):
    """Read filter information"""
    f = dict()
    f['freq'] = _read_double(fid)
    f['class'] = _read_int(fid)
    f['type'] = _read_int(fid)
    f['npar'] = _read_int2(fid)
    f['pars'] = [_read_double(fid) for _ in range(f['npar'])]
    return f


def _read_channel(fid):
    """Read channel information"""
    ch = dict()
    ch['sensor_type_index'] = _read_int2(fid)
    ch['original_run_no'] = _read_int2(fid)
    ch['coil_type'] = _read_int(fid)
    ch['proper_gain'] = _read_double(fid)
    ch['qgain'] = _read_double(fid)
    ch['io_gain'] = _read_double(fid)
    ch['io_offset'] = _read_double(fid)
    ch['num_coils'] = _read_int2(fid)
    ch['grad_order_no'] = int(_read_int2(fid))
    _read_int(fid)  # pad
    ch['coil'] = dict()
    ch['head_coil'] = dict()

    for coil in (ch['coil'], ch['head_coil']):
        coil['pos'] = list()
        coil['norm'] = list()
        coil['turns'] = np.empty(CTF.CTFV_MAX_COILS)
        coil['area'] = np.empty(CTF.CTFV_MAX_COILS)
        for k in range(CTF.CTFV_MAX_COILS):
            coil['pos'].append(np.array([_read_double(fid)
                                         for _ in range(3)]))
            # It would have been wonderful to use meters in the first place
            coil['pos'][-1] /= 100.0
            _read_double(fid)  # dummy
            coil['norm'].append(np.array([_read_double(fid)
                                          for _ in range(3)]))
            _read_double(fid)  # dummy
            coil['turns'][k] = _read_int2(fid)
            _read_int(fid)  # pad
            _read_int2(fid)  # pad
            # Looks like this is given in cm^2
            coil['area'][k] = _read_double(fid) * 1e-4
    return ch


def _read_comp_coeff(fid, d):
    """Read compensation coefficients"""
    # Read the coefficients and initialize
    d['ncomp'] = _read_int2(fid)
    d['comp'] = list()
    # Read each record
    for k in range(d['ncomp']):
        comp = dict()
        d['comp'].append(comp)
        comp['sensor_name'] = _read_string(fid, 32)
        comp['coeff_type'] = _read_int(fid)
        _read_int(fid)  # pad
        comp['ncoeff'] = _read_int2(fid)
        comp['coeffs'] = np.zeros(comp['ncoeff'])
        comp['sensors'] = list()
        for p in range(CTF.CTFV_MAX_BALANCING):
            sn = _read_string(fid, CTF.CTFV_SENSOR_LABEL, decode=False)
            comp['sensors'].append(sn.decode('utf-8')
                                   if p < comp['ncoeff'] else '')
        for p in range(CTF.CTFV_MAX_BALANCING):
            dval = _read_double(fid)
            if p < comp['ncoeff']:
                comp['coeffs'][p] = dval

        comp['scanno'] = -1
        for p in range(d['nchan']):
            if d['chs'][p]['ch_name'] == comp['sensor_name']:
                comp['scanno'] = p
                break


def _read_res4(dsdir):
    """Read the magical res4 file"""
    # adapted from read_res4.c
    name = _make_ctf_name(dsdir, 'res4')
    res = dict()
    with open(name, 'rb') as fid:
        # Read the fields
        res['head'] = _read_string(fid, 8)
        res['appname'] = _read_string(fid, 256)
        res['origin'] = _read_string(fid, 256)
        res['desc'] = _read_string(fid, 256)
        res['nave'] = _read_int2(fid)
        res['data_time'] = _read_string(fid, 255)
        res['data_date'] = _read_string(fid, 255)
        # Seems that date and time can be swapped
        # (are they entered manually?!)
        if '/' in res['data_time'] and ':' in res['data_date']:
            data_date = res['data_date']
            res['data_date'] = res['data_time']
            res['data_time'] = data_date
        res['nsamp'] = _read_int(fid)
        res['nchan'] = _read_int2(fid)
        _move_to_next(fid, 8)
        res['sfreq'] = _read_double(fid)
        res['epoch_time'] = _read_double(fid)
        res['no_trials'] = _read_int2(fid)
        _move_to_next(fid, 4)
        res['pre_trig_pts'] = _read_int(fid)
        res['no_trials_done'] = _read_int2(fid)
        res['no_trials_bst_message_windowlay'] = _read_int2(fid)
        _move_to_next(fid, 4)
        res['save_trials'] = _read_int(fid)
        res['primary_trigger'] = fid.read(1)
        res['secondary_trigger'] = [fid.read(1)
                                    for k in range(CTF.CTFV_MAX_AVERAGE_BINS)]
        res['trigger_polarity_mask'] = fid.read(1)
        res['trigger_mode'] = _read_int2(fid)
        _move_to_next(fid, 4)
        res['accept_reject'] = _read_int(fid)
        res['run_time_bst_message_windowlay'] = _read_int2(fid)
        _move_to_next(fid, 4)
        res['zero_head'] = _read_int(fid)
        _move_to_next(fid, 4)
        res['artifact_mode'] = _read_int(fid)
        _read_int(fid)  # padding
        res['nf_run_name'] = _read_string(fid, 32)
        res['nf_run_title'] = _read_string(fid, 256)
        res['nf_instruments'] = _read_string(fid, 32)
        res['nf_collect_descriptor'] = _read_string(fid, 32)
        res['nf_subject_id'] = _read_string(fid, 32)
        res['nf_operator'] = _read_string(fid, 32)
        res['nf_sensor_file_name'] = _read_ustring(fid, 60)
        _move_to_next(fid, 4)
        res['rdlen'] = _read_int(fid)
        fid.seek(CTF.FUNNY_POS, 0)

        if res['rdlen'] > 0:
            res['run_desc'] = _read_string(fid, res['rdlen'])

        # Filters
        res['nfilt'] = _read_int2(fid)
        res['filters'] = list()
        for k in range(res['nfilt']):
            res['filters'].append(_read_filter(fid))

        # Channel information
        res['chs'] = list()
        for k in range(res['nchan']):
            res['chs'].append(dict())
            res['chs'][k]['ch_name'] = _read_string(fid, 32)
        for k in range(res['nchan']):
            res['chs'][k].update(_read_channel(fid))

        # The compensation coefficients
        _read_comp_coeff(fid, res)
    logger.info('    res4 data read.')
    return res
